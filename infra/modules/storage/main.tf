# Lager en Storage Account som brukes til å lagre PDF-filer (Blob Storage)
resource "azurerm_storage_account" "sa" {
  name                = var.storage_account_name
  resource_group_name = var.resource_group_name
  location            = var.location
  # Standard SKU og replikering (nok for dette prosjektet)
  account_tier             = "Standard"
  account_replication_type = "LRS"

  # Konfigurasjon for Blob-tjenesten (f.eks. hvor lenge slettede blobs kan gjenopprettes)
  blob_properties {
    delete_retention_policy {
      days = 7
    }
  }

  tags = var.tags
}
# Egen container der vi legger alle PDF-filer
resource "azurerm_storage_container" "pdfs" {
  name                  = "pdfs"
  storage_account_id    = azurerm_storage_account.sa.id
  container_access_type = "private"
}
