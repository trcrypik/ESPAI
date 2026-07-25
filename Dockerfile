FROM nginx:alpine

# Копируем наш статический файл в дефолтную директорию Nginx
COPY index.html /usr/share/nginx/html/index.html

# Nginx по умолчанию слушает порт 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
