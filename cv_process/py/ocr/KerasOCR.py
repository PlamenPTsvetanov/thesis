class KerasOCR:
    @staticmethod
    def ocr_with_keras_ocr(region, pipeline):
        results = pipeline.recognize([region])
        text = KerasOCR.__get_formatted_text(results)
        return text, region

    @staticmethod
    def __get_formatted_text(results):
        text = ''
        for i in range(len(results[0])):
            text += ''.join(map(str, results[0][i][0]))

        return text.upper()
