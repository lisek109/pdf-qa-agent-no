output "resource_group_name" {
  value = module.container_app.resource_group_name
}

output "container_app_url" {
  description = "Basis-URL til PDF-assistenten i Azure Container Apps."
  value       = "https://${module.container_app.container_app_fqdn}"
}
