package bg.thesis.api.image;

import bg.thesis.api.base.BaseInView;
import bg.thesis.api.base.BaseRepository;
import bg.thesis.api.base.BaseService;
import bg.thesis.api.camera.CameraEntity;
import bg.thesis.api.camera.CameraInView;
import bg.thesis.api.camera.CameraOutView;
import bg.thesis.api.camera.CameraRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ImageService extends BaseService<ImageEntity, ImageOutView, BaseInView> {
    private final ImageRepository repository;

    @Autowired
    public ImageService(ImageRepository repository) {
        super(ImageEntity.class, ImageOutView.class);
        this.repository = repository;
    }

    @Override
    public BaseRepository<ImageEntity> getRepository() {
        return repository;
    }
}

