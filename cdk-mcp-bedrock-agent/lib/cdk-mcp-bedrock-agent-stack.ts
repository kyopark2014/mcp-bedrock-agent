import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as cloudFront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';

const projectName = `mcp-bedrock-agent`; 
const region = process.env.CDK_DEFAULT_REGION;    
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const bucketName = `storage-for-${projectName}-${accountId}-${region}`; 
const vectorIndexName = projectName

export class CdkMcpBedrockAgentStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Knowledge Base Role
    const knowledge_base_role = new iam.Role(this,  `role-knowledge-base-for-${projectName}`, {
      roleName: `role-knowledge-base-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("bedrock.amazonaws.com")
      )
    });
    
    const bedrockInvokePolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [
        `arn:aws:bedrock:*::foundation-model/*`
      ],
      actions: [
        "bedrock:InvokeModel", 
        "bedrock:Retrieve", 
        "bedrock:InvokeModelEndpoint", 
        "bedrock:InvokeModelEndpointAsync",        
      ],
    });        
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-invoke-policy-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    
    const bedrockKnowledgeBaseS3Policy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: [
        "s3:GetBucketLocation",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads",
        "s3:ListMultipartUploadParts",
        "s3:AbortMultipartUpload",
        "s3:CreateBucket",
        "s3:PutObject",
        "s3:PutBucketLogging",
        "s3:PutBucketVersioning",
        "s3:PutBucketNotification",
      ],
    });
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `knowledge-base-s3-policy-for-${projectName}`, {
        statements: [bedrockKnowledgeBaseS3Policy],
      }),
    );  
    
    const knowledgeBaseOpenSearchPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["aoss:APIAccessAll"],
    });
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-agent-opensearch-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );  

    const knowledgeBaseBedrockPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["bedrock:*"],
    });
    knowledge_base_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-agent-bedrock-policy-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  

    // OpenSearch Serverless
    const collectionName = vectorIndexName
    const OpenSearchCollection = new opensearchserverless.CfnCollection(this, `opensearch-correction-for-${projectName}`, {
      name: collectionName,    
      description: `opensearch correction for ${projectName}`,
      standbyReplicas: 'DISABLED',
      type: 'VECTORSEARCH',
    });
    const collectionArn = OpenSearchCollection.attrArn

    new cdk.CfnOutput(this, `OpensearchCollectionEndpoint-${projectName}`, {
      value: OpenSearchCollection.attrCollectionEndpoint,
      description: 'The endpoint of opensearch correction',
    });

    const encPolicyName = `encription-${projectName}`
    const encPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-encription-security-policy-for-${projectName}`, {
      name: encPolicyName,
      type: "encryption",
      description: `opensearch encryption policy for ${projectName}`,
      policy:
        `{"Rules":[{"ResourceType":"collection","Resource":["collection/${collectionName}"]}],"AWSOwnedKey":true}`,
    });
    OpenSearchCollection.addDependency(encPolicy);

    const netPolicyName = `network-${projectName}`
    const netPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-network-security-policy-for-${projectName}`, {
      name: netPolicyName,
      type: 'network',    
      description: `opensearch network policy for ${projectName}`,
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: "dashboard",
              Resource: [`collection/${collectionName}`],
            },
            {
              ResourceType: "collection",
              Resource: [`collection/${collectionName}`],              
            }
          ],
          AllowFromPublic: true,          
        },
      ]), 
      
    });
    OpenSearchCollection.addDependency(netPolicy);

    const account = new iam.AccountPrincipal(this.account)
    const dataAccessPolicyName = `data-${projectName}`
    const dataAccessPolicy = new opensearchserverless.CfnAccessPolicy(this, `opensearch-data-collection-policy-for-${projectName}`, {
      name: dataAccessPolicyName,
      type: "data",
      policy: JSON.stringify([
        {
          Rules: [
            {
              Resource: [`collection/${collectionName}`],
              Permission: [
                "aoss:CreateCollectionItems",
                "aoss:DeleteCollectionItems",
                "aoss:UpdateCollectionItems",
                "aoss:DescribeCollectionItems",
              ],
              ResourceType: "collection",
            },
            {
              Resource: [`index/${collectionName}/*`],
              Permission: [
                "aoss:CreateIndex",
                "aoss:DeleteIndex",
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadDocument",
                "aoss:WriteDocument",
              ], 
              ResourceType: "index",
            }
          ],
          Principal: [
            account.arn
          ], 
        },
      ]),
    });
    OpenSearchCollection.addDependency(dataAccessPolicy);

    // s3 
    const s3Bucket = new s3.Bucket(this, `storage-${projectName}`,{
      bucketName: bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      publicReadAccess: false,
      versioned: false,
      cors: [
        {
          allowedHeaders: ['*'],
          allowedMethods: [
            s3.HttpMethods.POST,
            s3.HttpMethods.PUT,
          ],
          allowedOrigins: ['*'],
        },
      ],
    });
    new cdk.CfnOutput(this, 'bucketName', {
      value: s3Bucket.bucketName,
      description: 'The nmae of bucket',
    });

    // agent role
    const agent_role = new iam.Role(this,  `role-agent-for-${projectName}`, {
      roleName: `role-agent-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("bedrock.amazonaws.com")
      )
    });

    const bedrockRetrievePolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [
        `arn:aws:bedrock:${region}:${accountId}:knowledge-base/*`
      ],
      actions: [
        "bedrock:Retrieve"
      ],
    });        
    agent_role.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-retrieve-policy-for-${projectName}`, {
        statements: [bedrockRetrievePolicy],
      }),
    );  
    
    const agentInferencePolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [
        `arn:aws:bedrock:${region}:${accountId}:inference-profile/*`,
        `arn:aws:bedrock:*::foundation-model/*`
      ],
      actions: [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetInferenceProfile",
        "bedrock:GetFoundationModel"
      ],
    });        
    agent_role.attachInlinePolicy( 
      new iam.Policy(this, `agent-inference-policy-for-${projectName}`, {
        statements: [agentInferencePolicy],
      }),
    );  

    // Lambda Invoke
    agent_role.addToPolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: [
        'lambda:InvokeFunction',
        'cloudwatch:*'
      ]
    }));
    agent_role.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/AWSLambdaExecute',
    });

    // Bedrock
    const BedrockPolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['bedrock:*'],
    });     
    agent_role.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `bedrock-policy-agent-for-${projectName}`, {
        statements: [BedrockPolicy],
      }),
    );   
    
    // Secret
    const weatherApiSecret = new secretsmanager.Secret(this, `weather-api-secret-for-${projectName}`, {
      description: 'secret for weather api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `openweathermap-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        weather_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    const langsmithApiSecret = new secretsmanager.Secret(this, `weather-langsmith-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `langsmithapikey-${projectName}`,
      secretObjectValue: {
        langchain_project: cdk.SecretValue.unsafePlainText(projectName),
        langsmith_api_key: cdk.SecretValue.unsafePlainText(''),
      }, 
    });

    const tavilyApiSecret = new secretsmanager.Secret(this, `weather-tavily-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `tavilyapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // Cost Explorer Policy
    const costExplorerPolicy = new iam.PolicyStatement({  
      resources: ['*'],
      actions: ['ce:GetCostAndUsage'],
    });        

    // cloudfront for sharing s3
    const distribution_sharing = new cloudFront.Distribution(this, `sharing-for-${projectName}`, {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200,  
    });
    new cdk.CfnOutput(this, `distribution-sharing-DomainName-for-${projectName}`, {
      value: 'https://'+distribution_sharing.domainName,
      description: 'The domain name of the Distribution Sharing',
    });        
    
    const mcp_config = JSON.stringify(`{
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": [
            "@playwright/mcp@latest"
          ]
        }
      }
    }`)
    const environment = {
      "projectName": projectName,
      "accountId": accountId,
      "region": region,
      "knowledge_base_role": knowledge_base_role.roleArn,
      "collectionArn": collectionArn,
      "opensearch_url": OpenSearchCollection.attrCollectionEndpoint,
      "s3_bucket": s3Bucket.bucketName,      
      "s3_arn": s3Bucket.bucketArn,
      "sharing_url": 'https://'+distribution_sharing.domainName,
      "mcp": mcp_config
    }    
    new cdk.CfnOutput(this, `environment-for-${projectName}`, {
      value: JSON.stringify(environment),
      description: `environment-${projectName}`,
      exportName: `environment-${projectName}`
    });
  }
}
