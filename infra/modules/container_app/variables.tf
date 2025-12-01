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
