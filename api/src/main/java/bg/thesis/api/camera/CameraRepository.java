package bg.thesis.api.camera;

import bg.thesis.api.base.BaseRepository;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface CameraRepository extends BaseRepository<CameraEntity> {
}
