package bg.thesis.api.image;

import bg.thesis.api.base.BaseOutView;
import jakarta.persistence.Column;
import jakarta.persistence.Lob;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.sql.Blob;

@Getter
@Setter
@ToString
public class ImageWithFilesOutView extends BaseOutView {
    @Column(name = "license_plate_image")
    @Lob
    private Blob licensePlateImage;

    @Column(name = "full_image")
    @Lob
    private Blob fullImage;
}
