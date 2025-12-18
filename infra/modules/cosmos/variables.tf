
variable "resource_group_name" {
  type        = string
  description = "Navn på RG for Cosmos DB"
}

variable "location" {
  type        = string
  description = "Azure-region"
}

variable "cosmos_account_name" {
  type        = string
  description = "Navn på Cosmos DB-konto"
}

variable "database_name" {
  type        = string
  description = "Navn på Cosmos database"
  default     = "pdfdb"
}

variable "container_name" {
  type        = string
  description = "Navn på Cosmos container"
  default     = "chunks"
}

variable "tags" {
  type    = map(string)
  default = {}
}
