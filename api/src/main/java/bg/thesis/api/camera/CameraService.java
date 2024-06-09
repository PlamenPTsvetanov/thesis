package bg.thesis.api.camera;

import bg.thesis.api.base.BaseRepository;
import bg.thesis.api.base.BaseService;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Component
public class CameraService extends BaseService<CameraEntity, CameraOutView, CameraInView> {
    @Autowired
    private CameraRepository repository;
    private static final Map<UUID, Process> processes = new HashMap<>();

    public CameraService() {
        super(CameraEntity.class, CameraOutView.class);
    }

    @Override
    public BaseRepository<CameraEntity> getRepository() {
        return repository;
    }


    @Transactional
    public CameraOutView setup(CameraInView cameraIn) throws ParserConfigurationException, TransformerException {
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.newDocument();

        Element rootElement = doc.createElement("camera");
        doc.appendChild(rootElement);

        Element name = doc.createElement("name");
        name.setTextContent(cameraIn.getName());
        rootElement.appendChild(name);

        Element imageFormat = doc.createElement("image_format");
        imageFormat.setTextContent(cameraIn.getImageFormat());
        rootElement.appendChild(imageFormat);

        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        Transformer transformer = transformerFactory.newTransformer();
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        DOMSource source = new DOMSource(doc);
        StreamResult result = new StreamResult(new File(cameraIn.getFolderPath() + File.separator + "camera_config.xml"));
        transformer.transform(source, result);

        CameraEntity entity = modelMapper.map(cameraIn, CameraEntity.class);
        getRepository().save(entity);

        return mapToOut(entity);
    }


    public void startProcess(UUID id, String ocr) throws IOException, InterruptedException {
        CameraEntity entity = getRepository().getReferenceById(id);

        String command = System.getProperty("user.dir") + "/api/command.sh";
        ProcessBuilder processBuilder = new ProcessBuilder("cmd", "/c", command, entity.getFolderPath(), ocr);

        Process process = processBuilder.start();
        processes.put(id, process);
        Thread.sleep(100);
        process.destroyForcibly();
    }
}

