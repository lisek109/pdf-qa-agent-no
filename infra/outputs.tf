output "resource_group_name" {
  value = module.container_app.resource_group_name
}

output "container_app_url" {
  description = "Basis-URL til Container App (uten revisjons-suffiks)."
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}
