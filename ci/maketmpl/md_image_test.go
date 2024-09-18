package maketmpl

import (
	"reflect"
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
