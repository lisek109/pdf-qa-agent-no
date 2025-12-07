output "resource_group_name" {
  value = azurerm_resource_group.rg.name
}

output "container_app_fqdn" {
  description = "Public URL til Container Appen"
  value       = azurerm_container_app.app.ingress[0].fqdn
}
