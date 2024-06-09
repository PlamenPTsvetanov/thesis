package bg.thesis.api.utils;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.MethodNotAllowedException;

import java.sql.Timestamp;
import java.time.LocalDateTime;
@RestControllerAdvice
public class CustomExceptionHandler {
    @ExceptionHandler({Exception.class})
    public ResponseEntity<CustomException> handleException(Exception e) {
        CustomException payload = new CustomException("An exception occurred!", e.getMessage(), Timestamp.valueOf(LocalDateTime.now()));

        return ResponseEntity.internalServerError().body(payload);
    }

    @ExceptionHandler({MethodNotAllowedException.class})
    public ResponseEntity<CustomException> handleNotAllowed(Exception e) {
        CustomException payload = new CustomException("Method is not allowed!", e.getMessage(), Timestamp.valueOf(LocalDateTime.now()));

        return ResponseEntity.internalServerError().body(payload);
    }
}

record CustomException(String userMessage, String systemMessage, Timestamp dateTime) {
}
