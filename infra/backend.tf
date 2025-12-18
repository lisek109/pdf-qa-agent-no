terraform {
  backend "azurerm" {
    resource_group_name  = "d-pdf-assistent-rg-1" # lub osobny RG na state, jeśli wolisz
    storage_account_name = "tisippdfstorage123"
    container_name       = "tfstate"
    key                  = "pdf-assistent/infra.tfstate"
  }
}
