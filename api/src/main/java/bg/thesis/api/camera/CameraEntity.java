package bg.thesis.api.camera;

import bg.thesis.api.base.BaseEntity;
import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.sql.Timestamp;

@Entity
@Table(name = "camera")
@Getter
@Setter
public class CameraEntity extends BaseEntity {
    @Column(name = "name")
    @NotNull(message = "Name cannot be null!")
    @NotBlank(message = "Name cannot be blank!")
    private String name;

    @Column(name = "folder_path")
    @NotNull(message = "Folder path cannot be null!")
    @NotBlank(message = "Folder path cannot be blank!")
    private String folderPath;

    @Column(name = "image_format")
    @NotNull(message = "Image format cannot be null!")
    @NotBlank(message = "Image format cannot be blank!")
    private String imageFormat;
}