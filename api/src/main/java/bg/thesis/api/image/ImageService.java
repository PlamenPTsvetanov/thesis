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

import java.sql.Time;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

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


    public List<ImageOutView> getFilteredOutput(UUID cameraId,
                                                String licensePlateNumber,
                                                String before,
                                                String after
    ) {
        List<ImageEntity> imageEntitiesFiltered = this.repository.getImageEntitiesFiltered(
                cameraId,
                licensePlateNumber,
                before == null ? null : Timestamp.valueOf(LocalDateTime.parse(before)),
                after == null ? null : Timestamp.valueOf(LocalDateTime.parse(after))
        );
        return mapList(imageEntitiesFiltered);
    }
}

