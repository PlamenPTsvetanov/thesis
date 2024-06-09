package bg.thesis.api.image;

import bg.thesis.api.base.BaseController;
import bg.thesis.api.base.BaseInView;
import bg.thesis.api.base.BaseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

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


    @GetMapping("/metadata")
    public ResponseEntity<List<ImageOutView>> getMetadata(@RequestParam(value = "cameraId", required = false) UUID cameraId,
                                                          @RequestParam(value = "number", required = false) String number,
                                                          @RequestParam(value = "before", required = false) @DateTimeFormat(pattern = "yyyy-mm-dd'T'hh:mm:ss") String before,
                                                          @RequestParam(value = "after", required = false) @DateTimeFormat(pattern = "yyyy-mm-dd'T'hh:mm:ss") String after) {
        return ResponseEntity.ok(this.service.getFilteredOutput(cameraId, number, before, after));
    }


    @GetMapping("/downloadImages")
    public ResponseEntity<byte[]> downloadImages(@RequestParam(value = "cameraId", required = false) UUID cameraId,
                                                 @RequestParam(value = "number", required = false) String number,
                                                 @RequestParam(value = "before", required = false) @DateTimeFormat(pattern = "yyyy-mm-dd'T'hh:mm:ss") String before,
                                                 @RequestParam(value = "after", required = false) @DateTimeFormat(pattern = "yyyy-mm-dd'T'hh:mm:ss") String after) throws IOException {
        List<ImageOutView> filteredOutput = this.service.getFilteredOutput(cameraId, number, before, after);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        try (ZipOutputStream zos = new ZipOutputStream(baos)) {
            int index = 1;
            for (ImageOutView image : filteredOutput) {
                if (Files.exists(Path.of(image.getFullImagePath()))) {
                    Path fullImagePath = Paths.get(image.getFullImagePath());
                    Path licensePath = Paths.get(image.getLicensePlateImagePath());

                    zos.putNextEntry(new ZipEntry(index + "_" + fullImagePath.getFileName().toString()));
                    Files.copy(fullImagePath, zos);
                    zos.closeEntry();

                    zos.putNextEntry(new ZipEntry(index++ + "_" + licensePath.getFileName().toString()));
                    Files.copy(licensePath, zos);
                    zos.closeEntry();
                }

            }
        } catch (IOException e) {
            return ResponseEntity.internalServerError().body(null);
        }

        baos.close();

        HttpHeaders headers = new HttpHeaders();
        headers.add("Content-Disposition", "attachment; filename=images.zip");
        return new ResponseEntity<>(baos.toByteArray(), headers, HttpStatus.OK);
    }
}
