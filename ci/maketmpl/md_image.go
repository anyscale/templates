package maketmpl

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"html"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
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

func inlineImgSrc(srcDir, src string) (string, error) {
	// We do not inline full URLs.
	if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
		return src, nil
	}

	var dataType string
	switch {
	case strings.HasSuffix(src, ".png"):
		dataType = "image/png"
	case strings.HasSuffix(src, ".jpg") || strings.HasSuffix(src, ".jpeg"):
		dataType = "image/jpeg"
	case strings.HasSuffix(src, ".gif"):
		dataType = "image/gif"
	default:
		return "", fmt.Errorf("unsupported image type: %s", src)
	}

	fp := filepath.Join(srcDir, src)
	imgData, err := os.ReadFile(fp)
	if err != nil {
		return "", fmt.Errorf("read image file: %w", err)
	}

	encoded := base64.StdEncoding.EncodeToString(imgData)
	return fmt.Sprintf("data:%s;base64,%s", dataType, encoded), nil
}

type writeImgOptions struct {
	inlineSrc   bool
	sizeInStyle bool
}

func (i *mdImage) writeInto(
	w io.Writer, srcDir string, opts *writeImgOptions,
) error {
	var styles []string
	if i.style != "" {
		styles = []string{i.style}
	}

	buf := new(bytes.Buffer)

	fmt.Fprint(buf, "<img ")
	if src := i.src; src != "" {
		if opts.inlineSrc {
			inlinedSrc, err := inlineImgSrc(srcDir, src)
			if err != nil {
				return fmt.Errorf("inline image: %w", err)
			}
			src = inlinedSrc
		}
		fmt.Fprintf(buf, `src="%s" `, html.EscapeString(src))
	}
	if v := i.alt; v != "" {
		fmt.Fprintf(buf, `alt="%s" `, html.EscapeString(v))
	}

	// add size. width first, height next.
	if v := i.widthPx; v != "" {
		if opts.sizeInStyle {
			styles = append(styles, fmt.Sprintf("width: %spx", i.widthPx))
		} else {
			fmt.Fprintf(buf, `width="%spx" `, html.EscapeString(v))
		}
	}
	if v := i.heightPx; v != "" {
		if opts.sizeInStyle {
			styles = append(styles, fmt.Sprintf("height: %spx", i.heightPx))
		} else {
			fmt.Fprintf(buf, `height="%spx" `, html.EscapeString(v))
		}
	}

	if len(styles) > 0 {
		style := strings.Join(styles, "; ")
		fmt.Fprintf(buf, `style="%s" `, html.EscapeString(style))
	}

	fmt.Fprint(buf, "/>")

	_, err := w.Write(buf.Bytes())
	return err
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

var (
	regexImageMd  = regexp.MustCompile(`!\[(.*)\]\((.*)\)`)
	regexImageTag = regexp.MustCompile(`<img src=.*>`)
)

func parseMdImages(md []byte) ([]*mdImage, error) {
	var imgs []*mdImage

	imgMds := regexImageMd.FindAllSubmatchIndex(md, -1)
	for _, found := range imgMds {
		imgs = append(imgs, &mdImage{
			start:  found[0],
			end:    found[1],
			src:    string(md[found[4]:found[5]]),
			alt:    string(md[found[2]:found[3]]),
			isHTML: false,
		})
	}

	imgTags := regexImageTag.FindAllIndex(md, -1)
	for _, found := range imgTags {
		s := string(md[found[0]:found[1]])
		img, err := paresImgHTMLTag(s)
		if err != nil {
			return nil, fmt.Errorf("parse img tag %q: %w", s, err)
		}
		img.start = found[0]
		img.end = found[1]
		imgs = append(imgs, img)
	}

	sort.Slice(imgs, func(i, j int) bool {
		if imgs[i].start == imgs[j].start {
			return imgs[i].end < imgs[j].end
		}
		return imgs[i].start < imgs[j].start
	})

	return imgs, nil
}
