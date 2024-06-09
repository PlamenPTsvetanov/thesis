package bg.thesis.api.camera;

import bg.thesis.api.base.BaseController;
import bg.thesis.api.base.BaseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.IOException;
import java.util.UUID;

@RestController
@RequestMapping(path = "/cameras")
public class CameraController extends BaseController<CameraEntity, CameraOutView, CameraInView> {

    @Autowired
    private CameraService service;

    @PostMapping("/setup")
    public ResponseEntity<CameraOutView> setupCamera(@RequestBody CameraInView cameraIn) throws ParserConfigurationException, TransformerException {
        CameraOutView payload = service.setup(cameraIn);

        return ResponseEntity.ok(payload);
    }

    @PostMapping("/start/{id}")
    public void startProcess(@PathVariable UUID id, @RequestParam String ocr) throws IOException, InterruptedException {
        service.startProcess(id, ocr);
    }

    @DeleteMapping("/kill/{id}")
    public void killProcess(@PathVariable UUID id) throws IOException {
        service.killProcess(id);
    }

    @Override
    protected BaseService<CameraEntity, CameraOutView, CameraInView> getService() {
        return service;
    }
}
