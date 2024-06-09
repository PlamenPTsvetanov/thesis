package bg.thesis.api.image;

import bg.thesis.api.base.BaseOutView;
import bg.thesis.api.camera.CameraEntity;
import bg.thesis.api.camera.CameraOutView;
import jakarta.persistence.Column;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.sql.Blob;
import java.util.UUID;

@Getter
@Setter
@ToString
public class ImageOutView extends BaseOutView {
    private String licensePlateNumber;

    private CameraOutView camera;
}
