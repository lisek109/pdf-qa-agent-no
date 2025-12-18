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
  source                 = "./modules/container_app"
  location               = var.location
  tags                   = local.common_tags
  rg_name                = local.rg_name
  log_name               = local.log_name
  cae_name               = local.cae_name
  container_app_name     = local.container_name
  container_image        = var.container_image
  blob_connection_string = module.storage.primary_connection_string
  cosmos_endpoint        = module.cosmos.cosmos_endpoint
  cosmos_key             = module.cosmos.cosmos_primary_key
  cosmos_db_name         = module.cosmos.database_name
  cosmos_container_name  = module.cosmos.container_name
}
