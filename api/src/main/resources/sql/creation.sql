create table camera
(
    id           uuid      default gen_random_uuid() not null
        constraint cameras_pkey
            primary key,
    name         text                                not null,
    folder_path  text                                not null,
    created_on   timestamp default now(),
    updated_on   timestamp,
    version      bigint    default 0                 not null,
    image_format text                                not null,
    constraint camera_uk
        unique (name, folder_path)
);

create table image
(
    id                       uuid      default gen_random_uuid() not null
        constraint images_pkey
            primary key,
    license_plate_number     varchar                             not null,
    license_plate_image_path text                                not null,
    full_image_path          text                                not null,
    camera_id                uuid                                not null
        constraint images_camera_fkey
            references camera,
    created_on               timestamp default now()             not null,
    updated_on               timestamp,
    version                  bigint    default 0                 not null
);