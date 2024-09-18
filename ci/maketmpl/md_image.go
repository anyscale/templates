package maketmpl

import (
	"io"
	"log"
	"strings"

	htmlx "golang.org/x/net/html"
)

type mdImage struct {
	start int
	end   int

	src string
	alt string

	heightPx string
	widthPx  string

	style string

	isHTML bool
}

func parsePxValue(s string) string {
	return strings.TrimSuffix(s, "px")
}

func paresImgHTMLTag(s string) (*mdImage, error) {
	if !strings.HasSuffix(s, "/>") {
		log.Printf("WARNING: image %q is not self-closing", s)
	}

	if strings.HasSuffix(s, "/>") && !strings.HasSuffix(s, " />") {
		// Insert a space to make it easier to parse
		s = s[:len(s)-2] + " />"
	}

	tok := htmlx.NewTokenizer(strings.NewReader(s))
	img := &mdImage{isHTML: true}
	for {
		t := tok.Next()
		if t == htmlx.StartTagToken || t == htmlx.SelfClosingTagToken {
			tagName, hasAttr := tok.TagName()
			if string(tagName) == "img" && hasAttr {
				for {
					k, v, more := tok.TagAttr()
					switch string(k) {
					case "src":
						img.src = string(v)
					case "alt":
						img.alt = string(v)
					case "height":
						img.heightPx = parsePxValue(string(v))
					case "width":
						img.widthPx = parsePxValue(string(v))
					case "style":
						img.style = string(v)
					}
					if !more {
						break
					}
				}
			}
		} else if t == htmlx.ErrorToken {
			err := tok.Err()
			if err == io.EOF {
				break
			}
			return nil, err
		}
	}

	return img, nil
}
