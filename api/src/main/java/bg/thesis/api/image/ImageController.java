package bg.thesis.api.image;

import bg.thesis.api.base.BaseController;
import bg.thesis.api.base.BaseInView;
import bg.thesis.api.base.BaseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping(path = "/images")
public class ImageController extends BaseController<ImageEntity, ImageOutView, BaseInView> {

    private final ImageService service;

    @Autowired
    public ImageController(ImageService service) {
        this.service = service;
    }

    @Override
    protected BaseService<ImageEntity, ImageOutView, BaseInView> getService() {
        return service;
    }
}
