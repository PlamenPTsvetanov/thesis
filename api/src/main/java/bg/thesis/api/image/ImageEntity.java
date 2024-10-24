package bg.thesis.api.image;

import bg.thesis.api.base.BaseEntity;
import bg.thesis.api.camera.CameraEntity;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.sql.Blob;
import java.util.UUID;

@Entity
@Table(name = "image")
@Getter
@Setter
public class ImageEntity extends BaseEntity {
    @Column(name = "license_plate_number")
    private String licensePlateNumber;

    @Column(name = "license_plate_image_path")
    private String licensePlateImagePath;

    @Column(name = "full_image_path")
    private String fullImagePath;

    @Column(name = "camera_id")
    private UUID cameraId;

    @ManyToOne
    @JoinColumn(name = "camera_id", updatable = false, insertable = false)
    private CameraEntity camera;
}