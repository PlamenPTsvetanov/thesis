package bg.thesis.api.base;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
public abstract class BaseController<Entity extends BaseEntity, OutView extends BaseOutView, InView extends BaseInView> {
    protected abstract BaseService<Entity, OutView, InView> getService();

    @GetMapping(path = "/all")
    public ResponseEntity<List<OutView>> getAll() {
        List<OutView> payload = this.getService().getAll();

        return ResponseEntity.ok(payload);
    }


    @GetMapping(path = "/{id}")
    public ResponseEntity<OutView> getOne(@PathVariable("id") UUID id) {
        OutView payload = this.getService().getOne(id);

        return ResponseEntity.ok(payload);
    }


    @PostMapping(path = "")
    public ResponseEntity<OutView> postOne(@RequestBody InView inView) {
        OutView payload = this.getService().postOne(inView);

        return ResponseEntity.ok(payload);
    }


    @PutMapping(path = "/{id}")
    public ResponseEntity<OutView> putOne(@PathVariable("id") UUID id, @RequestBody InView inView) {
        OutView payload = this.getService().putOne(id, inView);

        return ResponseEntity.ok(payload);
    }

    @DeleteMapping(path = "/{id}")
    public ResponseEntity<OutView> deleteOne(@PathVariable("id") UUID id) {
        OutView payload = this.getService().deleteOne(id);

        return ResponseEntity.ok(payload);
    }

}
