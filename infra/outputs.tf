output "resource_group_name" {
  value = module.container_app.resource_group_name
}

output "container_app_url" {
  value = "https://${module.container_app.container_app_fqdn}"
}
