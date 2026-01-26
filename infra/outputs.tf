output "bucket_name" {
  value = aws_s3_bucket.dvc_bucket.bucket
}

output "dvc_access_key" {
  value     = aws_iam_access_key.dvc_key.id
  sensitive = true  
}

output "dvc_secret_key" {
  value     = aws_iam_access_key.dvc_key.secret
  sensitive = true
}