import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError
from datetime import datetime

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug("Getting more packages for token: {}".format(response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            logger.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(f"Identified the latest approved model package: {model_package_arn}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def get_previous_model_name(project_name, endpoint_name):
    try:
        endpoint = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint["EndpointConfigName"]

        endpoint_config = sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )

        return endpoint_config["ProductionVariants"][0]["ModelName"]

    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        # Endpoint does not exist (first deployment)
        if error_code in ["ValidationException", "ResourceNotFound"]:
            logger.info(
                f"No existing endpoint found for {endpoint_name}. "
                "This is expected for first deployment."
            )
            return ""

        # Any other error is real and should fail the build
        raise


def extend_config(args, model_package_arn, stage_config):
    """
    Extend the stage configuration with additional parameters and tags based.
    """
    # Verify that config has parameters and tags sections
    if "Parameters" not in stage_config or "StageName" not in stage_config["Parameters"]:
        raise Exception("Configuration file must include SageName parameter")
    if "Tags" not in stage_config:
        stage_config["Tags"] = {}

    # Set model, endpoint configuration and endpoint names
    stage_name = stage_config['Parameters']['StageName']
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    candidate_model_name = f"Model-{args.sagemaker_project_name}-{stage_name}-{args.deployment_strategy}-{timestamp}"[:63]
    endpoint_config_name = f"EndpointConfig-{args.sagemaker_project_name}-{stage_name}-{timestamp}"
    endpoint_name = f"Endpoint-{args.sagemaker_project_name}-{stage_name}"

    # Get previous deployed model if exists
    previous_prod_model_name = get_previous_model_name(args.sagemaker_project_name, endpoint_name)
    if args.deployment_strategy not in ["first"] and not previous_prod_model_name:
        raise Exception(
            f"{args.deployment_strategy} requires an existing endpoint"
            f"Stage = {stage_config['Parameters']['StageName']}: No endpoint found."
        )

    # Create new params and tags
    new_params = {
        # General parameters
        "SageMakerProjectName": args.sagemaker_project_name,
        "DataCaptureUploadPath": "s3://" + args.s3_bucket + '/datacapture-' + stage_config["Parameters"]["StageName"],

        # Candidate model identifiers
        "ModelPackageName": model_package_arn,
        "ModelExecutionRoleArn": args.model_execution_role,

        # Previous and candidate model names
        "PreviousModelName": previous_prod_model_name,
        "CandidateModelName": candidate_model_name,

        # Endpoint metadata parameters
        "EndpointConfigName": endpoint_config_name,
        "EndpointName": endpoint_name,

        # Deployment strategy parameter
        "DeploymentStrategy": args.deployment_strategy,
    }

    # Add variant weights to parameters
    if args.deployment_strategy in ["canary", "ab"]:
        new_params["CandidateModelWeight"] = str(args.candidate_weight)
        new_params["PreviousModelWeight"] = str(1.0 - args.candidate_weight)
    else:
        new_params["CandidateModelWeight"] = str(args.candidate_weight)

    new_tags = {
        "sagemaker:deployment-stage": stage_config["Parameters"]["StageName"],
        "sagemaker:project-id": args.sagemaker_project_id,
        "sagemaker:project-name": args.sagemaker_project_name,
        "sagemaker:deployment-strategy": args.deployment_strategy
    }

    # Add tags from Project
    get_pipeline_custom_tags(args, sm_client, new_tags)

    return {
        "Parameters": {**stage_config["Parameters"], **new_params},
        "Tags": {**stage_config.get("Tags", {}), **new_tags},
    }


def get_pipeline_custom_tags(args, sm_client, new_tags):
    try:
        response = sm_client.describe_project(
            ProjectName=args.sagemaker_project_name
        )
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
                ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags[project_tag["Key"]] = project_tag["Value"]
    except:
        logger.error("Error getting project tags")
    return new_tags


def get_cfn_style_config(stage_config):
    parameters = []
    for key, value in stage_config["Parameters"].items():
        parameter = {
            "ParameterKey": key,
            "ParameterValue": value
        }
        parameters.append(parameter)
    tags = []
    for key, value in stage_config["Tags"].items():
        tag = {
            "Key": key,
            "Value": value
        }
        tags.append(tag)
    return parameters, tags


def create_cfn_params_tags_file(config, export_params_file, export_tags_file):
    # Write Params and tags in separate file for Cfn cli command
    parameters, tags = get_cfn_style_config(config)
    with open(export_params_file, "w") as f:
        json.dump(parameters, f, indent=4)
    with open(export_tags_file, "w") as f:
        json.dump(tags, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--model-execution-role", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--sagemaker-project-id", type=str, required=True)
    parser.add_argument("--sagemaker-project-name", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--deployment-strategy", type=str, default="single", choices=["first", "ab", "canary", "bluegreen", "shadow"])
    parser.add_argument("--candidate-weight", type=float, default=1.0)
    parser.add_argument("--import-staging-config", type=str, default="staging-config.json")
    parser.add_argument("--import-prod-config", type=str, default="prod-config.json")
    parser.add_argument("--export-staging-config", type=str, default="staging-config-export.json")
    parser.add_argument("--export-staging-params", type=str, default="staging-params-export.json")
    parser.add_argument("--export-staging-tags", type=str, default="staging-tags-export.json")
    parser.add_argument("--export-prod-config", type=str, default="prod-config-export.json")
    parser.add_argument("--export-prod-params", type=str, default="prod-params-export.json")
    parser.add_argument("--export-prod-tags", type=str, default="prod-tags-export.json")
    parser.add_argument("--export-cfn-params-tags", type=bool, default=False)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # Get the latest approved package
    logger.info("Getting latest approved model package.")
    model_package_arn = get_approved_package(args.model_package_group_name)
    logger.info(f"Latest approved model package ARN: {model_package_arn}")

    # Write the staging config
    with open(args.import_staging_config, "r") as f:
        staging_config = extend_config(args, model_package_arn, json.load(f))
    logger.debug("Staging config: {}".format(json.dumps(staging_config, indent=4)))
    with open(args.export_staging_config, "w") as f:
        json.dump(staging_config, f, indent=4)
    if (args.export_cfn_params_tags):
        create_cfn_params_tags_file(staging_config, args.export_staging_params, args.export_staging_tags)
    logger.info(f"Exported staging config with deployment strategy: {args.deployment_strategy}")

    # Write the prod config for code pipeline
    with open(args.import_prod_config, "r") as f:
        prod_config = extend_config(args, model_package_arn, json.load(f))
    logger.debug("Prod config: {}".format(json.dumps(prod_config, indent=4)))
    with open(args.export_prod_config, "w") as f:
        json.dump(prod_config, f, indent=4)
    if (args.export_cfn_params_tags):
        create_cfn_params_tags_file(prod_config, args.export_prod_params, args.export_prod_tags)
    logger.info(f"Exported production config with deployment strategy: {args.deployment_strategy}")
