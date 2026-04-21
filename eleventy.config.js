import { IdAttributePlugin, InputPathToUrlTransformPlugin, HtmlBasePlugin } from "@11ty/eleventy";
import pluginSyntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";
import pluginNavigation from "@11ty/eleventy-navigation";
import { eleventyImageTransformPlugin } from "@11ty/eleventy-img";

import markdownIt from "markdown-it";
import { katex } from "@mdit/plugin-katex";

import pluginFilters from "./_config/filters.js";

export default async function(eleventyConfig) {
	let options = {
		html: true,
		breaks: true,
		linkify: true,
	};

	const mdIt = markdownIt(options).use(katex);
	eleventyConfig.setLibrary("md", mdIt);

	// eleventyConfig.addPreprocessor("drafts", "*", (data, content) => {
	// 	if (data.draft) {
	// 		data.title = `${data.title} [draft]`;
	// 	}

	// 	if(data.draft && process.env.ELEVENTY_RUN_MODE === "build") {
	// 		return false;
	// 	}
	// });

	eleventyConfig.addPreprocessor("status", "*", (data, content) => {
		if (["todo", "wip"].includes(data.status)) {
			data.title = `${data.title} [${data.status}]`;
		}

		if(data.status == "todo" && process.env.ELEVENTY_RUN_MODE === "build") {
			return false;
		}
	});

	eleventyConfig
		.addPassthroughCopy({
			"./public/": "/"
		})

	eleventyConfig.addWatchTarget("css/**/*.css");
	eleventyConfig.addWatchTarget("content/**/*.{svg,webp,png,jpg,jpeg,gif}");

	eleventyConfig.addBundle("css", {
		toFileDirectory: "dist",
		bundleHtmlContentFromSelector: "style",
	});

	eleventyConfig.addBundle("js", {
		toFileDirectory: "dist",
		bundleHtmlContentFromSelector: "script",
	});

	eleventyConfig.addPlugin(pluginSyntaxHighlight, {
		preAttributes: { tabindex: 0 }
	});
	eleventyConfig.addPlugin(pluginNavigation);
	eleventyConfig.addPlugin(HtmlBasePlugin);
	eleventyConfig.addPlugin(InputPathToUrlTransformPlugin);

	eleventyConfig.addPlugin(eleventyImageTransformPlugin, {
		formats: ["avif", "webp", "auto"],
		// widths: ["auto"],

		failOnError: false,
		htmlOptions: {
			imgAttributes: {
				loading: "lazy",
				decoding: "async",
			}
		},

		sharpOptions: {
			animated: true,
		},
	});

	eleventyConfig.addPlugin(pluginFilters);
	eleventyConfig.addPlugin(IdAttributePlugin, {
		// slugify: eleventyConfig.getFilter("slugify"),
		// selector: "h1,h2,h3,h4,h5,h6", // default
	});

	eleventyConfig.addShortcode("currentBuildDate", () => {
		return (new Date()).toISOString();
	});

	eleventyConfig.addShortcode("plotly", function(data, layout = {}) {
        const id = `plot-${Math.random().toString(36).substring(2, 9)}`;
        
        return `
            <div id="${id}" style="width:100%; max-width:100%;"></div>
            <script type="module">
                import 'https://cdn.plot.ly/plotly-2.27.0.min.js';
                Plotly.newPlot('${id}', ${JSON.stringify(data)}, ${JSON.stringify(layout)});
            </script>
        `;
    });
};



export const config = {
	templateFormats: [
		"md",
		"njk",
		"html",
	],

	markdownTemplateEngine: "njk",
	htmlTemplateEngine: "njk",

	dir: {
		input: "content",
		includes: "../_includes",
		data: "../_data",
		output: "_site"
	},
};


