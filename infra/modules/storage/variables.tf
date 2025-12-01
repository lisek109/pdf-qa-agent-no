variable "resource_group_name" {
  description = "Navn på resource group der Storage Account skal ligge"
  type        = string
}

variable "location" {
  description = "Azure-region"
  type        = string
}

variable "storage_account_name" {
  description = "Unikt navn på Storage Account"
  type        = string
}

variable "tags" {
  description = "Tags som skal brukes på Storage-ressurser"
  type        = map(string)
  default     = {}
}
