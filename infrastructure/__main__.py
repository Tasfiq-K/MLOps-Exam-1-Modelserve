# ============================================================================
# ModelServe — Pulumi Infrastructure
# ============================================================================
# TODO: Provision the AWS resources your deployment topology requires.
#
# Your topology is YOUR decision. Common resources include:
#
#   Networking:
#     - VPC with a CIDR block
#     - Public subnet in an availability zone
#     - Internet gateway
#     - Route table with a default route to the internet gateway
#     - Route table association with the subnet
#
#   Security:
#     - Security group with ingress rules for your service ports
#     - Security group egress rule allowing all outbound traffic
#     - Consider: which ports actually need to be open? To whom?
#
#   Compute (if deploying to EC2):
#     - EC2 instance with appropriate instance type
#     - Key pair for SSH access
#     - Elastic IP for a stable address
#     - IAM instance profile + role with S3 and ECR permissions
#     - User-data script to install Docker and Docker Compose on boot
#
#   Storage:
#     - S3 bucket for MLflow artifacts and/or Feast offline store
#     - ECR repository for your Docker images (set force_delete=True)
#
# Requirements:
#   - All resources MUST be tagged with: Project = "modelserve"
#   - Export stack outputs for use by CI/CD (IPs, URLs, bucket names)
#   - pulumi destroy must cleanly remove everything
#   - Use os.environ.get("SSH_PUBLIC_KEY", "") for the key pair
#
# Refer to the Pulumi/CI/CD lab from Episodes 2-3 for patterns.
# ============================================================================


import os

import pulumi
import pulumi_aws as aws
from pulumi import Config, Output

# ============================================================================
# Config
# ============================================================================
unique_suffix = "main"
stack_name = pulumi.get_stack()
region = "ap-southeast-1"

ssh_public_key = os.environ.get("SSH_PUBLIC_KEY", "")
if not ssh_public_key:
    raise ValueError("SSH_PUBLIC_KEY environment variable is not set")

TAGS = {
    "Project": "modelserve",
    "Stack": stack_name,
}

# ============================================================================
# Key Pair
# ============================================================================
key = aws.ec2.KeyPair(
    "mlops-key",
    key_name=f"mlops-key-{unique_suffix}",
    public_key=ssh_public_key,
    tags=TAGS,
)

# ============================================================================
# Networking — VPC, subnet, IGW, route table
# ============================================================================
vpc = aws.ec2.Vpc(
    "mlops-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={"Name": f"mlops-vpc-{stack_name}", **TAGS},
)

igw = aws.ec2.InternetGateway(
    "mlops-igw",
    vpc_id=vpc.id,
    tags={"Name": f"mlops-igw-{stack_name}", **TAGS},
)

public_subnet = aws.ec2.Subnet(
    "mlops-public-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone=f"{region}a",
    map_public_ip_on_launch=True,
    tags={"Name": f"mlops-public-subnet-{stack_name}", **TAGS},
)

route_table = aws.ec2.RouteTable(
    "mlops-route-table",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=igw.id,
        )
    ],
    tags={"Name": f"mlops-route-table-{stack_name}", **TAGS},
)

aws.ec2.RouteTableAssociation(
    "mlops-rta",
    subnet_id=public_subnet.id,
    route_table_id=route_table.id,
)

# ============================================================================
# Security Group
# NOTE: All ports open to 0.0.0.0/0 — acceptable for dev/learning.
#       In production, restrict Prometheus/Grafana to your IP only.
# ============================================================================
security_group = aws.ec2.SecurityGroup(
    "mlops-sg",
    name=f"mlops-sg-{unique_suffix}",
    vpc_id=vpc.id,
    description="Security group for ModelServe services",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=22, to_port=22,
            cidr_blocks=["0.0.0.0/0"], description="SSH",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=8001, to_port=8001,
            cidr_blocks=["0.0.0.0/0"], description="FastAPI inference",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=8002, to_port=8002,
            cidr_blocks=["0.0.0.0/0"], description="Data ingestion service",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=9090, to_port=9090,
            cidr_blocks=["0.0.0.0/0"], description="Prometheus",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=3000, to_port=3000,
            cidr_blocks=["0.0.0.0/0"], description="Grafana",
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0,
            cidr_blocks=["0.0.0.0/0"], description="Allow all outbound",
        )
    ],
    tags={"Name": f"mlops-sg-{stack_name}", **TAGS},
)

# ============================================================================
# EC2 User Data — install Docker + Docker Compose on first boot
# ============================================================================
user_data = f"""#!/bin/bash
set -euo pipefail

apt-get update

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu
rm get-docker.sh

# Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# AWS CLI v2
apt-get install -y unzip curl jq
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Docker daemon config (log rotation for t2.micro)
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'DOCKER_EOF'
{{
  "log-driver": "json-file",
  "log-opts": {{
    "max-size": "10m",
    "max-file": "3"
  }},
  "storage-driver": "overlay2"
}}
DOCKER_EOF

systemctl enable docker
systemctl start docker

# Wait for Docker
timeout=60
while ! docker info > /dev/null 2>&1 && [ $timeout -gt 0 ]; do
    sleep 2
    timeout=$((timeout-2))
done

apt-get clean

echo "Setup complete at $(date)" > /home/ubuntu/setup-info.txt
chown ubuntu:ubuntu /home/ubuntu/setup-info.txt
"""

# ============================================================================
# EC2 Instance
# ============================================================================
instance = aws.ec2.Instance(
    "mlops-instance",
    key_name=key.key_name,
    instance_type="t2.micro",
    ami="ami-0df7a207adb9748c7",  # Ubuntu 22.04 LTS — ap-southeast-1
    subnet_id=public_subnet.id,
    vpc_security_group_ids=[security_group.id],
    user_data=user_data,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_type="gp2",
        volume_size=20,
        delete_on_termination=True,
    ),
    tags={"Name": f"mlops-instance-{stack_name}", **TAGS},
)

elastic_ip = aws.ec2.Eip(
    "mlops-eip",
    instance=instance.id,
    domain="vpc",
    tags={"Name": f"mlops-eip-{stack_name}", **TAGS},
)

# ============================================================================
# Stack Outputs — consumed by CI/CD
# ============================================================================
    pulumi.export("vpc_id", vpc.id)
    pulumi.export("subnet_id", public_subnet.id)
    pulumi.export("security_group_id", security_group.id)
    pulumi.export("instance_id", instance.id)
    pulumi.export("instance_public_ip", elastic_ip.public_ip)
    pulumi.export("grafana_url", Output.concat("http://", elastic_ip.public_ip, ":3000"))
    pulumi.export("prometheus_url", Output.concat("http://", elastic_ip.public_ip, ":9090"))
    pulumi.export("inference_url", Output.concat("http://", elastic_ip.public_ip, ":8001"))
    pulumi.export("unique_suffix", unique_suffix)