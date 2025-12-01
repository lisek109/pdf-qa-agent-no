# infra/locals.tf

locals {
  # Globalt navneprefix: <env-short>-<project>
  name_prefix = "${var.environment_short}-${var.project}"

  # Felles tags for alle ressurser
  common_tags = merge(
    { environment = var.environment },
    var.tags
  )

  # Ressursnavn for dette prosjektet
  rg_name        = "${local.name_prefix}-rg-${var.seq}"
  log_name       = "${local.name_prefix}-log-${var.seq}"
  cae_name       = "${local.name_prefix}-cae-${var.seq}"
  container_name = "${local.name_prefix}-ca-${var.seq}"
}
