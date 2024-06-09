package bg.thesis.api.base;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;
import jakarta.transaction.Transactional;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
public abstract class BaseService<Entity extends BaseEntity, OutView extends BaseOutView, InView extends BaseInView> {
    private final Class<OutView> outViewClass;
    private final Class<Entity> entityClass;
    @Autowired
    protected ModelMapper modelMapper;

    public abstract BaseRepository<Entity> getRepository();

    @PersistenceContext
    private EntityManager entityManager;


    public BaseService(Class<Entity> entityClass, Class<OutView> outViewClass) {
        this.outViewClass = outViewClass;
        this.entityClass = entityClass;
    }

    public List<OutView> getAll() {
        List<Entity> all = this.getRepository().findAll();
        return mapList(all);
    }

    public OutView getOne(UUID id) {
        Entity one = this.getRepository().getReferenceById(id);
        return mapToOut(one);
    }

    @Transactional
    public OutView postOne(InView inView) {
        Entity post = mapToEntity(inView);
        Entity save = this.getRepository().save(post);

        entityManager.flush();
        entityManager.refresh(save);

        return mapToOut(save);
    }

    @Transactional
    public OutView putOne(UUID id, InView inView) {
        Entity data = this.getRepository().getReferenceById(id);

        modelMapper.map(inView, data);

        Entity save = this.getRepository().save(data);

        entityManager.flush();
        entityManager.refresh(save);
        return mapToOut(save);
    }

    @Transactional
    public OutView deleteOne(UUID id) {
        Entity data = this.getRepository().getReferenceById(id);

        this.getRepository().deleteById(id);

        return mapToOut(data);
    }

    /*----------------------------- Helper methods -----------------------------*/
    protected List<OutView> mapList(List<Entity> entities) {
        List<OutView> outViewList = new ArrayList<>();
        for (Entity entity : entities) {
            outViewList.add(modelMapper.map(entity, outViewClass));
        }
        return outViewList;
    }

    protected OutView mapToOut(Entity entity) {
        return modelMapper.map(entity, outViewClass);
    }

    protected Entity mapToEntity(InView inView) {
        return modelMapper.map(inView, entityClass);
    }

}
