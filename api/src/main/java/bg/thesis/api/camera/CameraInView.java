package bg.thesis.api.camera;

import bg.thesis.api.base.BaseInView;
import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.sql.Timestamp;

@Getter
@Setter
@ToString
public class CameraInView extends BaseInView {
    @NotNull(message = "Name cannot be null!")
    @NotBlank(message = "Name cannot be blank!")
    private String name;
    @NotNull(message = "Folder path cannot be null!")
    @NotBlank(message = "Folder path cannot be blank!")
    private String folderPath;
    @NotNull(message = "Image format cannot be null!")
    @NotBlank(message = "Image format cannot be blank!")
    private String imageFormat;
    @JsonIgnore
    private Timestamp updatedOn;
}
