import { IdAttributePlugin, InputPathToUrlTransformPlugin, HtmlBasePlugin } from "@11ty/eleventy";
import pluginSyntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";
import pluginNavigation from "@11ty/eleventy-navigation";
import { eleventyImageTransformPlugin } from "@11ty/eleventy-img";

import markdownIt from "markdown-it";
import { katex } from "@mdit/plugin-katex";
import { container } from "@mdit/plugin-container";

import pluginFilters from "./_config/filters.js";

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

export default async function(eleventyConfig) {

	const mdIt = markdownIt({
		html: true,
		breaks: true,
		linkify: true,
	}).use(
		katex
	).use(container, {
		name: "foldable",
		validate: function(params) {
			return params.trim().match(/^\s*foldable\s*(.*)\s*$/);
		},
		openRender: function(tokens, idx) {
			const m = tokens[idx].info.trim().match(/^\s*foldable\s*(.*)\s*$/);
			const title = m[1] ? mdIt.utils.escapeHtml(m[1]) : "Details";
			return `<details><summary> ${title} </summary>\n`;
		},
		closeRender: function(tokens, idx) {
			return "</details>\n";
		}
	});

	eleventyConfig.setLibrary("md", mdIt);

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
			"./public/": "/",
		})

	eleventyConfig.addWatchTarget("css/**/*.css");
	eleventyConfig.addWatchTarget("js/**/*.js");
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
		widths: ["auto"],

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
		slugify: eleventyConfig.getFilter("slugify"),
		selector: "h1,h2,h3,h4,h5,h6",
	});

	eleventyConfig.addShortcode("currentBuildDate", () => {
		return (new Date()).toISOString();
	});

	eleventyConfig.addShortcode("plotly", function(relDataFile, params = {}) {
		const inputDataFile = path.resolve(path.dirname(this.page.inputPath), relDataFile);
		const outputDir = path.dirname(this.page.outputPath);
		const outputUrl = path.dirname(this.page.url);

		const data = fs.readFileSync(inputDataFile);
		const id = crypto.hash("md5", data);

		return `
			<div 
				id="${id}" 
				class="plotly-container" 
				data-plotly="${encodeURIComponent(data)}"
				data-title="${params.title || ''}"
				data-type="${params.type || 'scatter'}"
				data-smooth="${params.smooth || 'false'}"
			>
			</div>
		`.trim();
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


