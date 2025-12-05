# Ressursgruppe
resource "azurerm_resource_group" "rg" {
  name     = var.rg_name
  location = var.location
  tags     = var.tags
}

# Log Analytics
resource "azurerm_log_analytics_workspace" "law" {
  name                = var.log_name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

# Container Apps Environment
resource "azurerm_container_app_environment" "env" {
  name                       = var.cae_name
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.law.id
}

# Selve Container Appen (kjører Docker-imaet )
resource "azurerm_container_app" "app" {
  name                         = var.container_app_name
  resource_group_name          = azurerm_resource_group.rg.name
  container_app_environment_id = azurerm_container_app_environment.env.id
  revision_mode                = "Single"

  template {
    container {
      name   = "pdf-assistent"
      image  = var.container_image
      cpu    = 0.5
      memory = "1Gi"

      # --- Backend-modus: lokal eller sky ---
      env {
        name  = "BACKEND_MODE"
        value = "cloud"
      }

      # --- Blob Storage connection string ---
      env {
        name = "BLOB_CONNECTION_STRING"
        # Verdien kommer fra storage-modulen (primary_connection_string)
        value = var.blob_connection_string
      }

      # --- Cosmos DB-tilkobling ---
      env {
        name = "COSMOS_ENDPOINT"
        # Basis-URL til Cosmos-kontoen 
        value = var.cosmos_endpoint
      }

      env {
        name = "COSMOS_KEY"
        # Primærnøkkel for Cosmos DB (brukes for å autentisere fra Python)
        value = var.cosmos_key
      }

      env {
        name = "COSMOS_DB"
        # Navn på databasen der vi lagrer dokumenter/chunks (f.eks. 'pdfdb')
        value = var.cosmos_db_name
      }

      env {
        name = "COSMOS_CONTAINER"
        # Navn på containeren som inneholder chunks med embeddings fe  'chunks'
        value = var.cosmos_container_name
      }
    }

    # MIN/MAX 
    min_replicas = 0
    max_replicas = 1
  }

  ingress {
    external_enabled = true
    target_port      = 8501

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  tags = var.tags
}
