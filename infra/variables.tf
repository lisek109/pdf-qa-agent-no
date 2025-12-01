# infra/variables.tf

variable "project" {
  description = "Prosjektnavn (kort)"
  type        = string
  default     = "pdf"
}

variable "environment" {
  description = "Miljø (f.eks. dev, test, prod)"
  type        = string
  default     = "dev"
}

variable "environment_short" {
  description = "Kort miljøkode (d, t, p)"
  type        = string
  default     = "d"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "westeurope"
}

variable "resource_group_name" {
  description = "Navn på resource group for PDF-assistenten"
  type        = string
  default     = "rg-pdf-assistent"
}

variable "storage_account_name" {
  description = "Unikt navn på Storage Account"
  type        = string
}

variable "cosmos_account_name" {
  description = "Navn på Cosmos DB-konto"
  type        = string
}

variable "container_image" {
  description = "Fullt Docker image navn for PDF-assistenten"
  type        = string
}

variable "seq" {
  description = "Sekvensnummer for å sikre unike navn"
  type        = number
  default     = 1
}

variable "tags" {
  description = "Ekstra tags for alle ressurser"
  type        = map(string)
  default     = {}
}

