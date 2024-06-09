package bg.thesis.api.camera;

import bg.thesis.api.base.BaseOutView;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
public class CameraOutView extends BaseOutView {
    private String name;
    private String folderPath;
}
