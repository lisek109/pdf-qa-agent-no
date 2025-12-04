terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  # For dette skoleprosjektet hardkoder vi subscription_id,
  # slik at Terraform vet nøyaktig hvilken Azure-subscription som skal brukes.
  # I et produksjonsoppsett ville dette normalt kommet fra:
  # - 'az account set' (CLI-kontekst), eller
  # - miljøvariabler (ARM_SUBSCRIPTION_ID, ARM_TENANT_ID, osv.).
  subscription_id = "7a3c6854-0fe1-42eb-b5b9-800af1e53d70"
}
