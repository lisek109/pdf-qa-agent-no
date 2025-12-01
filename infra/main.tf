# infra/main.tf


module "storage" {
  source = "./modules/storage"

  resource_group_name  = module.container_app.resource_group_name
  location             = var.location
  storage_account_name = var.storage_account_name
  tags                 = local.common_tags
}

module "cosmos" {
  source = "./modules/cosmos"

  resource_group_name = module.container_app.resource_group_name
  location            = var.location
  cosmos_account_name = var.cosmos_account_name
  tags                = local.common_tags
}

module "container_app" {
  source             = "./modules/container_app"
  location           = var.location
  tags               = local.common_tags
  rg_name            = local.rg_name
  log_name           = local.log_name
  cae_name           = local.cae_name
  container_app_name = local.container_name
  container_image    = var.container_image

  # Her kan du senere også sende inn:
  # - connection strings til Blob / Cosmos
  # - BACKEND_MODE ("cloud")
}
