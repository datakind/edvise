GRANT USE CATALOG ON CATALOG `{{DB_workspace}}` TO `{{app_service_principal}}`;

GRANT USE SCHEMA ON SCHEMA `{{DB_workspace}}`.`default` TO `{{app_service_principal}}`;

GRANT SELECT ON TABLE `{{DB_workspace}}`.`default`.`pipeline_runs` TO `{{app_service_principal}}`;

GRANT SELECT ON TABLE `{{DB_workspace}}`.`default`.`pipeline_models` TO `{{app_service_principal}}`;
