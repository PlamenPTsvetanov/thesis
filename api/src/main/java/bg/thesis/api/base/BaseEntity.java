package bg.thesis.api.base;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.math.BigInteger;
import java.sql.Timestamp;
import java.util.UUID;

@Getter
@Setter
@MappedSuperclass
@EntityListeners(AuditingEntityListener.class)
public abstract class BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private UUID id;

    @CreatedDate
    @Column(name = "created_on")
    private Timestamp createdOn;

    @LastModifiedDate
    @Column(name = "updated_on")
    private Timestamp updatedOn;

    @Version
    private Long version;
}
