resource "azurerm_storage_account" "sa" {
  name                     = var.storage_account_name
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  kind                     = "StorageV2"

  allow_blob_public_access = false

  tags = var.tags
}

resource "azurerm_storage_container" "pdfs" {
  name                  = "pdfs"
  storage_account_name  = azurerm_storage_account.sa.name
  container_access_type = "private"
}
