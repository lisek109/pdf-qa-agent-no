output "storage_account_name" {
  value = azurerm_storage_account.sa.name
}

output "pdf_container_name" {
  value = azurerm_storage_container.pdfs.name
}

output "primary_connection_string" {
  # Denne kan brukes i appen for å koble til Blob Storage
  value     = azurerm_storage_account.sa.primary_connection_string
  sensitive = true
}
