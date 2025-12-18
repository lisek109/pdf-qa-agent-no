# Oppretter Cosmos DB-konto (NoSQL API)
resource "azurerm_cosmosdb_account" "cosmos" {
  name                = var.cosmos_account_name
  location            = var.location
  resource_group_name = var.resource_group_name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB" # NoSQL API

  consistency_policy {
    consistency_level = "Session"
  }

  capabilities {
    name = "EnableAggregationPipeline"
  }

  capabilities {
    name = "EnableServerless"
  }

  geo_location {
    location          = var.location
    failover_priority = 0
  }

  tags = var.tags
}

# Cosmos database
resource "azurerm_cosmosdb_sql_database" "db" {
  name                = var.database_name
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.cosmos.name
}

# Cosmos container (vektor + metadata per chunk)
resource "azurerm_cosmosdb_sql_container" "container" {
  name                = var.container_name
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.cosmos.name
  database_name       = azurerm_cosmosdb_sql_database.db.name

  # Viktig: userId blir partisjonsnøkkel → perfekt for multi-tenant opersjoner i apapen
  partition_key_paths = ["/userId"]

  # Indexpolicy – kreves for vector search (Terraform støtter ikke full vector-index ennå, men policy kan oppdateres senere fra Azure Portal)
  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }
  }
}
