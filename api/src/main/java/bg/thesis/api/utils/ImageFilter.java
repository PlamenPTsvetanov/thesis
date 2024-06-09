package bg.thesis.api.utils;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
public class ImageFilter extends OncePerRequestFilter {
    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
        if (request.getRequestURL().toString().contains("images") && !request.getMethod().equals("GET")) {
            response.sendError(HttpServletResponse.SC_METHOD_NOT_ALLOWED, "Only GET method is allowed for /images!");

            return;
        }
        filterChain.doFilter(request, response);
    }
}
