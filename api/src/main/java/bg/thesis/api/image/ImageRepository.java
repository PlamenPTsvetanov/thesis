package bg.thesis.api.image;

import bg.thesis.api.base.BaseRepository;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Repository
public interface ImageRepository extends BaseRepository<ImageEntity> {

    @Query("""
            SELECT i FROM ImageEntity i
            WHERE (:cameraId IS NULL OR i.cameraId = :cameraId)
            AND (:licensePlateNumber IS NULL OR i.licensePlateNumber = :licensePlateNumber)
            AND (cast(:before as timestamp ) IS NULL OR i.createdOn < :before)
            AND (cast(:after as timestamp) IS NULL OR i.createdOn >= :after)
            """)
    List<ImageEntity> getImageEntitiesFiltered(@Param("cameraId") UUID cameraId,
                                               @Param("licensePlateNumber") String licensePlateNumber,
                                               @Param("before") Timestamp before,
                                               @Param("after") Timestamp after);
}
