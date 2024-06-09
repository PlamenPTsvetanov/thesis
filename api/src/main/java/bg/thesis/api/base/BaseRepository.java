package bg.thesis.api.base;

import com.fasterxml.jackson.databind.ser.Serializers;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;

public interface BaseRepository<Entity extends BaseEntity>
        extends JpaRepository<Entity, UUID> {

}
