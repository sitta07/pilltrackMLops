terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-southeast-1" # สิงคโปร์ (ใกล้ไทยสุด)
}

# ==========================================
# 1. สร้าง S3 Bucket
# ==========================================
resource "aws_s3_bucket" "dvc_bucket" {
  bucket = "pilltrack-mlops-storage" 
  
  tags = {
    Project     = "PillTrack"
    Environment = "Dev"
  }
}

# เปิด Versioning (เผื่อ DVC พัง เรากู้ไฟล์เก่าได้)
resource "aws_s3_bucket_versioning" "dvc_versioning" {
  bucket = aws_s3_bucket.dvc_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ==========================================
# 2. สร้าง IAM User (สำหรับ DVC)
# ==========================================
resource "aws_iam_user" "dvc_user" {
  name = "dvc-pilltrack-bot"
}

# สร้าง Access Key ให้ User นี้
resource "aws_iam_access_key" "dvc_key" {
  user = aws_iam_user.dvc_user.name
}

# ==========================================
# 3. สร้าง IAM Policy (สิทธิ์การเข้าถึง)
# ==========================================
resource "aws_iam_user_policy" "dvc_policy" {
  name = "dvc-s3-access-policy"
  user = aws_iam_user.dvc_user.name

  # Policy: ให้ทำได้ทุกอย่าง "เฉพาะใน Bucket นี้เท่านั้น"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:ListBucketMultipartUploads"
        ]
        Effect   = "Allow"
        Resource = aws_s3_bucket.dvc_bucket.arn
      },
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject",
          "s3:ListMultipartUploadParts",
          "s3:AbortMultipartUpload"
        ]
        Effect   = "Allow"
        Resource = "${aws_s3_bucket.dvc_bucket.arn}/*"
      }
    ]
  })
}