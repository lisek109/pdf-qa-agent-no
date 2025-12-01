variable "location" {
  type = string
}

variable "tags" {
  type = map(string)
}

variable "rg_name" {
  type = string
}

variable "log_name" {
  type = string
}

variable "cae_name" {
  type = string
}

variable "container_app_name" {
  type = string
}

variable "container_image" {
  type = string
}

# Tilkoblingstreng til Blob Storage (kommer fra storage-modulen)
variable "blob_connection_string" {
  type        = string
  description = "Connection string for Blob Storage (brukes i appen)"
}

# Cosmos-tilkobling (kommer fra cosmos-modulen)
variable "cosmos_endpoint" {
  type        = string
  description = "Cosmos DB endpoint URL"
}

variable "cosmos_key" {
  type        = string
  description = "Primærnøkkel for Cosmos DB"
  sensitive   = true
}

variable "cosmos_db_name" {
  type        = string
  description = "Navn på Cosmos-databasen"
}

variable "cosmos_container_name" {
  type        = string
  description = "Navn på Cosmos-containeren for chunks"
}
