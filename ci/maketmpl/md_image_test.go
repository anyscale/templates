package maketmpl

import (
	"reflect"
	"strings"
	"testing"
)

func TestParsePxValue(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{in: "400px", want: "400"},
		{in: "400", want: "400"},
		{in: "400.5px", want: "400.5"},
		{in: "0", want: "0"},
		{in: "0px", want: "0"},
		{in: "-400px", want: "-400"},
	}

	for _, test := range tests {
		got := parsePxValue(test.in)
		if got != test.want {
			t.Errorf("parsePxValue(%q) = %q, want %q", test.in, got, test.want)
		}
	}
}

func TestParseImgHTMLTag(t *testing.T) {
	tests := []struct {
		in   string
		want *mdImage
	}{{
		in:   `<img src=img.png />`,
		want: &mdImage{src: "img.png", isHTML: true},
	}, {
		in: `<img src="img.png" height="400" />`,
		want: &mdImage{
			src: "img.png", heightPx: "400", isHTML: true,
		},
	}, {
		in: `<img src="./a.png" alt=alt width=700px/>`,
		want: &mdImage{
			src: "./a.png", alt: "alt", widthPx: "700", isHTML: true,
		},
	}, {
		in:   `<img alt="&quot;a">`,
		want: &mdImage{alt: `"a`, isHTML: true},
	}, {
		in: `<img src="a.jpeg" style="height: 300px; width: 500px"/>`,
		want: &mdImage{
			src:    "a.jpeg",
			style:  "height: 300px; width: 500px",
			isHTML: true,
		},
	}, {
		// concating "px" and "/>"
		// this is technically invalid html, but we should handle it.
		in: `<img src="a.jpeg" width=300px/>`,
		want: &mdImage{
			src: "a.jpeg", widthPx: "300",
			isHTML: true,
		},
	}}

	for _, test := range tests {
		got, err := paresImgHTMLTag(test.in)
		if err != nil {
			t.Errorf("parseImgHTMLTag(%q): %+v", test.in, err)
			continue
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf(
				"parseImgHTMLTag(%q), got %+v, want %+v",
				test.in, got, test.want,
			)
		}
	}
}

func TestParseMdImages(t *testing.T) {
	md := strings.Join([]string{
		"![first image](img.png)",
		"some random text",
		`<img src=img2.png alt="second image" width=200px/>`,
		"some more random text",
		"![third image](img3.png)",
	}, "\n")

	imgs, err := parseMdImages([]byte(md))
	if err != nil {
		t.Fatalf("parseMdImages: %v", err)
	}

	want := []*mdImage{
		{src: "img.png", alt: "first image"},
		{src: "img2.png", alt: "second image", widthPx: "200", isHTML: true},
		{src: "img3.png", alt: "third image"},
	}

	wantLens := []int{23, 50, 24}

	if len(imgs) != len(want) {
		t.Errorf("got %d images, want %d", len(imgs), len(want))
	} else {
		for i, img := range imgs {
			want := want[i]
			if img.src != want.src {
				t.Errorf("img %d, got src %q, want %q", i, img.src, want.src)
			}
			if img.alt != want.alt {
				t.Errorf("img %d, got alt %q, want %q", i, img.alt, want.alt)
			}
			if img.widthPx != want.widthPx {
				t.Errorf(
					"img %d, got width %q, want %q",
					i, img.widthPx, want.widthPx,
				)
			}
			if img.isHTML != want.isHTML {
				t.Errorf(
					"img %d, got isHTML %t, want %t",
					i, img.isHTML, want.isHTML,
				)
			}

			lenOfImg := img.end - img.start
			if lenOfImg != wantLens[i] {
				t.Errorf(
					"img %d, got length %d, want %d",
					i, lenOfImg, wantLens[i],
				)
			}
		}

		if imgs[0].start != 0 {
			t.Errorf("first image start is %d, want 0", imgs[0].start)
		}
	}
}
